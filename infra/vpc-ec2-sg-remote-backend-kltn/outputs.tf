

output "instance_id" {
  value       = module.ec2.instance_id
  description = "The ID of the EC2 instance."

}

output "instance_public_ip" {
  value       = module.ec2.instance_public_ip
  description = "The public IP address of the EC2 instance."

}